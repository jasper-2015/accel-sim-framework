// Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../ISA_Def/ampere_opcode.h"
#include "../ISA_Def/kepler_opcode.h"
#include "../ISA_Def/pascal_opcode.h"
#include "../ISA_Def/trace_opcode.h"
#include "../ISA_Def/turing_opcode.h"
#include "../ISA_Def/volta_opcode.h"
#include "../ISA_Def/accelwattch_component_mapping.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu_context.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "trace_driven.h"

const trace_warp_inst_t *trace_shd_warp_t::get_next_trace_inst() {
  if (trace_pc < warp_traces.size()) {
    trace_warp_inst_t *new_inst =
        new trace_warp_inst_t(get_shader()->get_config());
    new_inst->parse_from_trace_struct(
        warp_traces[trace_pc], m_kernel_info->OpcodeMap,
        m_kernel_info->m_tconfig, m_kernel_info->m_kernel_trace_info);
    trace_pc++;
    return new_inst;
  } else
    return NULL;
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  warp_traces.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() { return trace_pc == (warp_traces.size()); }

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].m_pc;
}

address_type trace_shd_warp_t::get_pc() {
  assert(warp_traces.size() > 0);
  assert(trace_pc < warp_traces.size());
  return warp_traces[trace_pc].m_pc;
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_function_info *m_function_info,
                                         trace_parser *parser,
                                         class trace_config *config,
                                         kernel_trace_t *kernel_trace_info,
                                         unsigned int appwin, unsigned int kerwin,
                                         unsigned int ker_local_win,
                                         unsigned int max_win_single)
    : kernel_info_t(gridDim, blockDim, m_function_info) {
  m_parser = parser;
  m_tconfig = config;
  m_kernel_trace_info = kernel_trace_info;
  m_was_launched = false;
  m_appwin = appwin;
  m_kerwin = kerwin;
  m_ker_local_win = ker_local_win;
  m_max_win_single = max_win_single;

  // resolve the binary version
  if (kernel_trace_info->binary_verion == AMPERE_RTX_BINART_VERSION ||
      kernel_trace_info->binary_verion == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
           kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           kernel_trace_info->binary_verion);
    fflush(stdout);
    exit(0);
  }
}

void trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
  m_parser->get_next_threadblock_traces(threadblock_traces,
                                        m_kernel_trace_info->trace_verion,
                                        m_kernel_trace_info->ifs);
}

types_of_operands get_oprnd_type(op_type op, special_ops sp_op){
  switch (op) {
    case SP_OP:
    case SFU_OP:
    case SPECIALIZED_UNIT_2_OP:
    case SPECIALIZED_UNIT_3_OP:
    case DP_OP:
    case LOAD_OP:
    case STORE_OP:
      return FP_OP;
    case INTP_OP:
    case SPECIALIZED_UNIT_4_OP:
      return INT_OP;
    case ALU_OP:
      if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
        return FP_OP;
      else if (sp_op == INT__OP)
        return INT_OP;
    default: 
      return UN_OP;
  }
}

bool trace_warp_inst_t::parse_from_trace_struct(
    const inst_trace_t &trace,
    const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
    const class trace_config *tconfig,
    const class kernel_trace_t *kernel_trace_info) {
  // fill the inst_t and warp_inst_t params

  // fill active mask
  active_mask_t active_mask = trace.mask;
  set_active(active_mask);

  // fill and initialize common params
  m_decoded = true;
  pc = (address_type)trace.m_pc;
  a_pc = (address_type)trace.a_pc;

    // Ni
  if (trace.mem_local_reg == 1) {
    mem_local_reg = true;
  }
  else if (trace.mem_local_reg == 0) {
    mem_local_reg = false;
  }
  else {
    printf("inst pc with no mem_local_reg: 0x%llx\n", pc);
    fflush(stdout);
    abort();  // Can only be 1 or 0
  }

  isize =
      16;  // starting from MAXWELL isize=16 bytes (including the control bytes)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }

  is_vectorin = 0;
  is_vectorout = 0;
  ar1 = 0;
  ar2 = 0;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  sp_op = OTHER_OP;
  mem_op = NOT_TEX;
  const_cache_operand = 0;
  oprnd_type = UN_OP;

  // get the opcode
  std::vector<std::string> opcode_tokens = trace.get_opcode_tokens();
  std::string opcode1 = opcode_tokens[0];

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);
  if (it != OpcodeMap->end()) {
    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
    const std::unordered_map<unsigned, unsigned> *OpcPowerMap = &OpcodePowerMap;
    std::unordered_map<unsigned, unsigned>::const_iterator it2 =
      OpcPowerMap->find(m_opcode);
    if(it2 != OpcPowerMap->end())
      sp_op = (special_ops) (it2->second);
      oprnd_type = get_oprnd_type(op, sp_op);
  } else {
    std::cout << "ERROR:  undefined instruction : " << trace.opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }
  std::string opcode = trace.opcode;
  if(opcode1 == "MUFU"){ // Differentiate between different MUFU operations for power model
    if ((opcode == "MUFU.SIN") || (opcode == "MUFU.COS"))
      sp_op = FP_SIN_OP;
    if ((opcode == "MUFU.EX2") || (opcode == "MUFU.RCP"))
      sp_op = FP_EXP_OP;
    if (opcode == "MUFU.RSQ") 
      sp_op = FP_SQRT_OP;
    if (opcode == "MUFU.LG2") 
      sp_op = FP_LG_OP;
  }

  if(opcode1 == "IMAD"){ // Differentiate between different IMAD operations for power model
    if ((opcode == "IMAD.MOV") || (opcode == "IMAD.IADD"))
      sp_op = INT__OP;
  }
  
  // fill regs information
  num_regs = trace.reg_srcs_num + trace.reg_dsts_num;
  num_operands = num_regs;
  outcount = trace.reg_dsts_num;
  for (unsigned m = 0; m < trace.reg_dsts_num; ++m) {
    out[m] =
        trace.reg_dest[m] + 1;  // Increment by one because GPGPU-sim starts
                                // from R1, while SASS starts from R0
    arch_reg.dst[m] = trace.reg_dest[m] + 1;
  }

  incount = trace.reg_srcs_num;
  for (unsigned m = 0; m < trace.reg_srcs_num; ++m) {
    in[m] = trace.reg_src[m] + 1;  // Increment by one because GPGPU-sim starts
                                   // from R1, while SASS starts from R0
    arch_reg.src[m] = trace.reg_src[m] + 1;
  }

  // fill latency and initl
  tconfig->set_latency(op, latency, initiation_interval);

  // fill addresses
  if (trace.memadd_info != NULL) {
    data_size = trace.memadd_info->width;
    for (unsigned i = 0; i < warp_size(); ++i)
      set_addr(i, trace.memadd_info->addrs[i]);
  }


  // handle special cases and fill memory space
  switch (m_opcode) {
    case OP_LDC: //handle Load from Constant
      data_size = 4;
      memory_op = memory_load;
      const_cache_operand = 1;
      space.set_type(const_space);
      cache_op = CACHE_ALL;
      break;
    case OP_LDG:
    case OP_LDL:
      assert(data_size > 0);
      memory_op = memory_load;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_LDL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      // check the cache scope, if its strong GPU, then bypass L1
      if (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
          trace.check_opcode_contain(opcode_tokens, "GPU")) {
        cache_op = CACHE_GLOBAL;
      }
      break;
    case OP_STG:
    case OP_STL:
      assert(data_size > 0);
      memory_op = memory_store;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_STL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      break;
    case OP_ATOMG:
    case OP_RED:
    case OP_ATOM:
      assert(data_size > 0);
      memory_op = memory_load;
      op = LOAD_OP;
      space.set_type(global_space);
      m_isatomic = true;
      cache_op = CACHE_GLOBAL;  // all the atomics should be done at L2
      break;
    case OP_LDS:
      assert(data_size > 0);
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_STS:
      assert(data_size > 0);
      memory_op = memory_store;
      space.set_type(shared_space);
      break;
    case OP_ATOMS:
      assert(data_size > 0);
      m_isatomic = true;
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_LDSM:
      assert(data_size > 0);
      space.set_type(shared_space);
      break;
    case OP_ST:
    case OP_LD:
      assert(data_size > 0);
      if (m_opcode == OP_LD)
        memory_op = memory_load;
      else
        memory_op = memory_store;
      // resolve generic loads
      if (kernel_trace_info->shmem_base_addr == 0 ||
          kernel_trace_info->local_base_addr == 0) {
        // shmem and local addresses are not set
        // assume all the mem reqs are shared by default
        space.set_type(shared_space);
      } else {
        // check the first active address
        for (unsigned i = 0; i < warp_size(); ++i)
          if (active_mask.test(i)) {
            if (trace.memadd_info->addrs[i] >=
                    kernel_trace_info->shmem_base_addr &&
                trace.memadd_info->addrs[i] <
                    kernel_trace_info->local_base_addr)
              space.set_type(shared_space);
            else if (trace.memadd_info->addrs[i] >=
                         kernel_trace_info->local_base_addr &&
                     trace.memadd_info->addrs[i] <
                         kernel_trace_info->local_base_addr +
                             LOCAL_MEM_SIZE_MAX) {
              space.set_type(local_space);
              cache_op = CACHE_ALL;
            } else {
              space.set_type(global_space);
              cache_op = CACHE_ALL;
            }
            break;
          }
      }

      break;
    case OP_BAR:
      // TO DO: fill this correctly
      bar_id = 0;
      bar_count = (unsigned)-1;
      bar_type = SYNC;
      // TO DO
      // if bar_type = RED;
      // set bar_type
      // barrier_type bar_type;
      // reduction_type red_type;
      break;
    case OP_HADD2:
    case OP_HADD2_32I:
    case OP_HFMA2:
    case OP_HFMA2_32I:
    case OP_HMUL2_32I:
    case OP_HSET2:
    case OP_HSETP2:
      initiation_interval =
          initiation_interval / 2;  // FP16 has 2X throughput than FP32
      break;
    case OP_CALL:
      m_funwin = trace.funwin;
      m_depwin = trace.depwin;
      m_is_relo_call = trace.is_relo_call;
      break;
    case OP_RET:
      m_funwin = trace.funwin;
      m_depwin = trace.depwin;
      break;
    default:
      break;
  }

  return true;
}

trace_config::trace_config() {}

void trace_config::reg_options(option_parser_t opp) {
  option_parser_register(opp, "-trace", OPT_CSTR, &g_traces_filename,
                         "traces kernel file"
                         "traces kernel file directory",
                         "./traces/kernelslist.g");

  option_parser_register(opp, "-trace_opcode_latency_initiation_int", OPT_CSTR,
                         &trace_opcode_latency_initiation_int,
                         "Opcode latencies and initiation for integers in "
                         "trace driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sp", OPT_CSTR,
                         &trace_opcode_latency_initiation_sp,
                         "Opcode latencies and initiation for sp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_dp", OPT_CSTR,
                         &trace_opcode_latency_initiation_dp,
                         "Opcode latencies and initiation for dp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sfu", OPT_CSTR,
                         &trace_opcode_latency_initiation_sfu,
                         "Opcode latencies and initiation for sfu in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_tensor",
                         OPT_CSTR, &trace_opcode_latency_initiation_tensor,
                         "Opcode latencies and initiation for tensor in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-reg_win_mode",
                         OPT_UINT32, &reg_win_mode,
                         "Reg window mode "
                         "1 for app_win, 2 for ker_win, 3 for fun_win",
                         "0");
  option_parser_register(opp, "-lowmark_window_multiple",
                         OPT_UINT32, &lowmark_window_multiple,
                         "# of lowmark window size allocated for each warp",
                         "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-trace_opcode_latency_initiation_spec_op_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &trace_opcode_latency_initiation_specialized_op[j],
                           "specialized unit config"
                           " <latency,initiation>",
                           "4,4");
  }
}

void trace_config::parse_config() {
  sscanf(trace_opcode_latency_initiation_int, "%u,%u", &int_latency, &int_init);
  sscanf(trace_opcode_latency_initiation_sp, "%u,%u", &fp_latency, &fp_init);
  sscanf(trace_opcode_latency_initiation_dp, "%u,%u", &dp_latency, &dp_init);
  sscanf(trace_opcode_latency_initiation_sfu, "%u,%u", &sfu_latency, &sfu_init);
  sscanf(trace_opcode_latency_initiation_tensor, "%u,%u", &tensor_latency,
         &tensor_init);

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    sscanf(trace_opcode_latency_initiation_specialized_op[j], "%u,%u",
           &specialized_unit_latency[j], &specialized_unit_initiation[j]);
  }
}
void trace_config::set_latency(unsigned category, unsigned &latency,
                               unsigned &initiation_interval) const {
  initiation_interval = latency = 1;

  switch (category) {
    case ALU_OP:
    case INTP_OP:
    case BRANCH_OP:
    case CALL_OPS:
    case RET_OPS:
      latency = int_latency;
      initiation_interval = int_init;
      break;
    case SP_OP:
      latency = fp_latency;
      initiation_interval = fp_init;
      break;
    case DP_OP:
      latency = dp_latency;
      initiation_interval = dp_init;
      break;
    case SFU_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      break;
    case TENSOR_CORE_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      break;
    default:
      break;
  }
  // for specialized units
  if (category >= SPEC_UNIT_START_ID) {
    unsigned spec_id = category - SPEC_UNIT_START_ID;
    assert(spec_id >= 0 && spec_id < SPECIALIZED_UNIT_NUM);
    latency = specialized_unit_latency[spec_id];
    initiation_interval = specialized_unit_initiation[spec_id];
  }
}

void trace_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new trace_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                    m_shader_stats, m_memory_stats);
}

void trace_simt_core_cluster::create_shader_core_ctx() {
  m_core = new shader_core_ctx *[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}

void trace_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new trace_shd_warp_t(this, m_config->warp_size);
  }
}

void trace_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                    const warp_inst_t *pI,
                                                    unsigned *pc,
                                                    unsigned *rpc) {
  // In trace-driven mode, we assume no control hazard
  // if (pI) {
    *pc = pI->pc;
    *rpc = pI->pc;
  // }
}

const active_mask_t &trace_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  // For Trace-driven, the active mask already set in traces, so
  // just read it from the inst
  return pI->get_active_mask();
}

bool trace_shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM

  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    trace_kernel_info_t &trace_kernel = static_cast<trace_kernel_info_t &>(kernel);
    if (trace_kernel.m_tconfig->reg_win_mode == 4) {
      // Use block scheduling or not is not decided
      return (get_n_active_cta() < m_config->max_cta(kernel));
    } else {
      return (get_n_active_cta() < m_config->max_cta(kernel));
    }
  }
}

unsigned trace_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (kernel.more_threads_in_cta()) {
    kernel.increment_thread_id();
  }

  if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

  return 1;
}

void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size, kernel_info_t &kernel) {
  // call base class
  shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size,
                              kernel);

  // then init traces
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  init_traces(start_warp, end_warp, kernel);
}

const warp_inst_t *trace_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                        address_type pc) {
  // read the inst from the traces
  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  trace_kernel_info_t *kernel_info = static_cast<trace_kernel_info_t *>(m_trace_warp->get_kernel_info());

  // Used for 5
  if (kernel_info->m_tconfig->reg_win_mode == 5) {
    const warp_inst_t* next_inst = m_trace_warp->get_next_trace_inst();
    if (m_alloc_fail_record[warp_id] != 0 || !next_inst || !next_inst->mem_local_reg) {
      return next_inst;
    }
    else {
      do {
        next_inst = m_trace_warp->get_next_trace_inst();
        if (!next_inst) {
          break;
        }
      } while (next_inst->mem_local_reg);
      return next_inst;
    }
  }

  if (kernel_info->m_tconfig->reg_win_mode == 7) {
    // TODO
    if (m_spf_to_execute[warp_id].size() > 0) {
      if (warp_id == WID && get_sid() == SID) {
        printf("size of spf list: %u\n", m_spf_to_execute[warp_id].size());
        fflush(stdout);
      }
    
      // inst_trace_t* spf_inst = new inst_trace_t();
      // assert(spf_inst->parse_from_string(*m_spf_to_execute[warp_id].front(), 3));
      // trace_warp_inst_t *new_inst =
      //   new trace_warp_inst_t(m_config);
      // new_inst->parse_from_trace_struct(
      //     *spf_inst, kernel_info->OpcodeMap,
      //     kernel_info->m_tconfig, kernel_info->m_kernel_trace_info);
      // delete spf_inst;
      
      delete []m_spf_to_execute[warp_id].front();
      m_spf_to_execute[warp_id].pop();
      
      // return new_inst;
    }
    // else {
      // if (m_spf_to_execute[warp_id].size() == 0) {
      //   printf("clear the spf queue\n");
      //   fflush(stdout);
      //   m_spf_to_execute[warp_id].clear();
      // }
      const warp_inst_t* next_inst;
      do {
        next_inst = m_trace_warp->get_next_trace_inst();
        if (!next_inst) {
          break;
        }
      } while (next_inst->mem_local_reg);
      return next_inst;
    // }
  }

  // Used for 6, 8, 10
  else if (kernel_info->m_tconfig->reg_win_mode == 6 || kernel_info->m_tconfig->reg_win_mode == 8
            || kernel_info->m_tconfig->reg_win_mode == 10) {
    const warp_inst_t* next_inst = m_trace_warp->get_next_trace_inst();
    if ((m_pair_record[warp_id].size() != 0 && m_pair_record[warp_id].back()) || !next_inst || !next_inst->mem_local_reg) {
      return next_inst;
    }
    else {
      do {
        next_inst = m_trace_warp->get_next_trace_inst();
        if (!next_inst) {
          break;
        }
      } while (next_inst->mem_local_reg);
      return next_inst;
    }
  }

  // Used for 9
  // const warp_inst_t* next_inst = NULL;
  // if (m_spf_to_execute[warp_id].size() > 0) {
  //   if (m_spf_to_execute[warp_id].front() != NULL) {
  //     next_inst = m_spf_to_execute[warp_id].front();
  //     // printf("next_inst %llx\n", next_inst->a_pc);
  //   }
  //   // if (next_inst == NULL) {
  //   //   printf("size of m_spf_to_execute: %u\n", m_spf_to_execute[warp_id].size());
  //   //   printf("inst is null on warp %u shader %u\n", warp_id, get_sid());
  //   // }
  //   m_spf_to_execute[warp_id].erase(m_spf_to_execute[warp_id].begin());
  // }
  // if (next_inst == NULL) {
  //   do {
  //     next_inst = m_trace_warp->get_next_trace_inst();
  //     if (!next_inst) {
  //       break;
  //     }
  //     // const trace_warp_inst_t *pI = static_cast<const trace_warp_inst_t *>(next_inst);
  //     // if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
  //     if (next_inst->op == CALL_OPS) {
  //       reg_window new_reg_window;
  //       new_reg_window.in_reg = true;
  //       new_reg_window.borrowed = -1;
  //       new_reg_window.used_own = 0;
  //       m_window_per_warp[warp_id].push_back(new_reg_window);
  //       break;
  //     }
  //     else if (next_inst->mem_local_reg) {
        
  //       // warp_inst_t* next_inst_copy = new warp_inst_t();
  //       // *next_inst_copy = *next_inst;
  //       // memcpy(next_inst_copy, next_inst, sizeof(warp_inst_t));
        
  //       if (next_inst->is_load()) {
  //         // printf("LOAD: copy start!\n");
  //         m_window_per_warp[warp_id].back().ldl.push_back(next_inst);
  //         // m_window_per_warp[warp_id].back().ldl.push_back(next_inst_copy);
  //         // printf("LOAD:copy finish!\n");
  //         fflush(stdout);
  //       }
  //       else if (next_inst->is_store()) {
  //         // printf("STORE:copy start!\n");
  //         m_window_per_warp[warp_id].back().stl.push_back(next_inst);
  //         // m_window_per_warp[warp_id].back().stl.push_back(next_inst_copy);
  //         // printf("STORE:copy finish!\n");
  //         fflush(stdout);
  //       }
  //       else {
  //         assert(false);
  //       }
  //     }
  //   } while (next_inst->mem_local_reg);
  //   // printf("next_inst %llx\n", next_inst->a_pc);
  // }
  // return next_inst;

  else {
    return m_trace_warp->get_next_trace_inst();
  }
}

void trace_shader_core_ctx::updateSIMTStack(unsigned warpId,
                                            warp_inst_t *inst) {
  // No SIMT-stack in trace-driven  mode
}

void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        kernel_info_t &kernel) {
  std::vector<std::vector<inst_trace_t> *> threadblock_traces;
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->clear();
    threadblock_traces.push_back(&(m_trace_warp->warp_traces));
    m_dep_table[i][0] = 1;
    m_dep_table[i][1] = 0;
    m_dep_table[i][2] = 1;
  }
  trace_kernel_info_t &trace_kernel =
      static_cast<trace_kernel_info_t &>(kernel);
  trace_kernel.get_next_threadblock_traces(threadblock_traces);

  // Ni: get the static #reg/warp based on the nreg info in the trace file
  kernel_trace_t* kernel_info = trace_kernel.get_trace_info();
  static_reg_per_warp = 0;
  num_warps_per_cta = (kernel_info->tb_dim_x * kernel_info->tb_dim_y * kernel_info->tb_dim_z)
                              / 32;
  if (num_warps_per_cta == 0) {
    num_warps_per_cta = 1; // total # of threads is less than 32
  }
  static_reg_per_warp = (m_config->gpgpu_shader_registers / 32) / (kernel_max_cta_per_shader * num_warps_per_cta);
  
  // unsigned max_cta_per_core = 2048 / (kernel_info->nregs * num_warps_per_cta);
  // unsigned total_cta = kernel_info->grid_dim_x * kernel_info->grid_dim_y * kernel_info->grid_dim_z;
  // if (total_cta >= max_cta_per_core) {
  //   static_reg_per_warp = 2048 / (max_cta_per_core * num_warps_per_cta);
  // }
  // else {
  //   static_reg_per_warp = 2048 / (total_cta * num_warps_per_cta);
  // }

  // num_reg_left = (m_config->gpgpu_shader_registers / 32) - ((kernel_info->ker_local_win + 16) * kernel_max_cta_per_shader * num_warps_per_cta);
  num_reg_left = m_config->gpgpu_shader_registers / 32;
  unsigned num_running_warp = kernel_max_cta_per_shader * num_warps_per_cta;
  if (num_reg_left < 0) {
    num_running_warp = 0;
  }
  else {
    if (trace_kernel.m_tconfig->reg_win_mode == 8) {
      if (kernel_info->max_ker_win > 0) {
        num_running_warp = num_reg_left / (kernel_info->max_ker_win + kernel_info->ker_local_win + 32);
      }
      else {
        num_running_warp = kernel_max_cta_per_shader * num_warps_per_cta;
      }
    }
    else if (trace_kernel.m_tconfig->reg_win_mode == 10) {
      if (kernel_info->max_reg_win_single > (kernel_info->ker_local_win + 16)) {
        num_running_warp = num_reg_left / (kernel_info->max_reg_win_single + 16);
      }
      else {
        num_running_warp = num_reg_left / (kernel_info->ker_local_win + 32);
      }
    }
    if (num_running_warp > kernel_max_cta_per_shader * num_warps_per_cta) {
      num_running_warp = kernel_max_cta_per_shader * num_warps_per_cta;
    }
  }
  
  if (num_running_warp == 0) {
    num_running_warp = 1;
    stall_all = true;
  }
  else {
    stall_all = false;
  }
  assert(!stall_all);

  if (kernel_info->bar) {
    if (end_warp <= num_running_warp) {
      for (unsigned i = start_warp; i < end_warp; i++) {
        m_warp_stall[i] = false;
      }
    }
    else {
      for (unsigned i = start_warp; i < end_warp; i++) {
        m_warp_stall[i] = true;
        if (get_sid() == SID)
          printf("stall warp %u\n", i);
      }
    }
  }
  
  else {
    for (unsigned i = start_warp; i < end_warp; i++) {
      if (i < num_running_warp) {
        m_warp_stall[i] = false;
      }
      else {
        m_warp_stall[i] = true;
        if (get_sid() == SID)
          printf("stall warp %u\n", i);
      }
    }
  }

  // Used for 5, 7, 9
  if (trace_kernel.m_tconfig->reg_win_mode == 5 || trace_kernel.m_tconfig->reg_win_mode == 7 || 
      trace_kernel.m_tconfig->reg_win_mode == 9) {
    for (unsigned i = start_warp; i < end_warp; i++) {
      m_free_reg_per_warp[i] = static_reg_per_warp;
    }
  } 

  // Used for 10
  if (trace_kernel.m_tconfig->reg_win_mode == 10) {
    for (unsigned i = start_warp; i < end_warp; i++) {
      if (stall_all) {
        m_free_reg_per_warp[i] = num_reg_left;
        abort();
      }
      else {
        if (kernel_info->max_reg_win_single > (kernel_info->ker_local_win + 16)) {
          m_free_reg_per_warp[i] = kernel_info->max_reg_win_single + 16;
        }
        else {
          m_free_reg_per_warp[i] = kernel_info->ker_local_win + 32;
        }
        if ((num_reg_left / num_running_warp) > m_free_reg_per_warp[i]) {
          m_free_reg_per_warp[i] = num_reg_left / num_running_warp;
        }
        m_free_reg_per_warp[i] -= (kernel_info->ker_local_win + 32);
      }
    }
    if (get_sid() == 0) {
      printf("reg: %u\n", m_free_reg_per_warp[start_warp]);
      fflush(stdout);
    }
  } 

  // Used for 8
  if (trace_kernel.m_tconfig->reg_win_mode == 8) {
    for (unsigned i = start_warp; i < end_warp; i++) {
      // m_free_reg_per_warp[i] = num_reg_left;
      m_free_reg_per_warp[i] = kernel_info->max_ker_win;
    }
  }

  // Used for 9
  // for (unsigned i = start_warp; i < end_warp; i++) {
  //   m_free_reg_per_warp[i] = static_reg_per_warp - (kernel_info->ker_local_win + 16);
  //   reg_window new_reg_window;
  //   new_reg_window.reg_num = kernel_info->ker_local_win + 16;
  //   new_reg_window.in_reg = true;
  //   new_reg_window.borrowed = -1;
  //   new_reg_window.used_own = 0;
  //   m_window_per_warp[i].push_back(new_reg_window);
  // }
  
  // num_chunk_per_warp = static_reg_per_warp / m_config->gpgpu_regchunk;
  // unsigned num_chunk_left = (2048 / m_config->gpgpu_regchunk) - (num_chunk_per_warp * kernel_max_cta_per_shader * num_warps_per_cta);
  // m_freelist_start = (2048 / m_config->gpgpu_regchunk) - num_chunk_left;
  // for (unsigned i = m_freelist_start; i < (2048 / m_config->gpgpu_regchunk); i++) {
  //   m_freelist.push_back(i);
  // }

  // for (unsigned k = 0; k < m_warp_regchunk.size(); k++) {
  //     m_warp_regchunk[k] = -1; // Empty
  //   }
  
  if (get_sid() == 0) {
  //   printf("start_warp: %u, end_warp: %u\n", start_warp, end_warp);
    printf("kernel_max_cta_per_shader: %u\n", kernel_max_cta_per_shader);
    // assert(kernel_max_cta_per_shader * num_warps_per_cta * 32 == 2048);
    printf("static_reg_per_warp: %u\n", static_reg_per_warp);
    printf("num_running_warp: %u\n", num_running_warp);
    printf("num_reg_left: %u\n", num_reg_left);
    // for (unsigned i = 0; i < m_freelist.size(); i++) {
    //   printf("m_freelist[%u]: %u\n", i, m_freelist[i]);
    // }
  }
  fflush(stdout);

  // set the pc from the traces and ignore the functional model
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
    m_trace_warp->set_kernel(&trace_kernel);
  }
}

void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                          unsigned t,
                                                          unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();

  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (new_addr_type *)localaddrs);
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
  }

  if (inst.op == EXIT_OPS) {
    m_warp[inst.warp_id()]->set_completed(t);
  }
}

void trace_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (inst.active(t)) {
      unsigned warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }

  // here, we generate memory acessess and set the status if thread (done?)
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
  }

  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[inst.warp_id()]);
  if (m_trace_warp->trace_done() && m_trace_warp->functional_done()) {
    m_trace_warp->ibuffer_flush();
    m_barriers.warp_exit(inst.warp_id());
    m_dep_table[inst.warp_id()][0] = 0;
    m_dep_table[inst.warp_id()][1] = 0;
    m_dep_table[inst.warp_id()][2] = 1;
    m_dep_table[inst.warp_id()][3] = 0;

    m_free_reg_per_warp[inst.warp_id()] = static_reg_per_warp;

    if (m_call_record[inst.warp_id()].size() != 0) {
      printf("call_record incorrect\n");
      printf("inst pc 0x%llx on warp %u and SM %u\n", inst.pc, inst.warp_id(), get_sid());
    }
    assert(m_call_record[inst.warp_id()].size() == 0);
    m_call_record[inst.warp_id()].clear();

    if (m_pair_record[inst.warp_id()].size() != 0) {
      printf("call_record incorrect\n");
      printf("inst pc 0x%llx on warp %u and SM %u\n", inst.pc, inst.warp_id(), get_sid());
    }
    assert(m_pair_record[inst.warp_id()].size() == 0);
    m_pair_record[inst.warp_id()].clear();
    m_start_ker[inst.warp_id()] = false;
    m_ker_fail[inst.warp_id()] = false;

    m_reg_stack_map[inst.warp_id()].clear();
    // m_spf_to_execute[inst.warp_id()].clear();
    m_win_record[inst.warp_id()].clear();

    printf("total_func_call: %u\n", total_func_call[inst.warp_id()]);
    printf("func_call_spf: %u\n", func_call_spf[inst.warp_id()]);
    printf("byte_spf: %llu\n", byte_spf[inst.warp_id()]);

    total_func_call[inst.warp_id()] = 0;
    func_call_spf[inst.warp_id()] = 0;
    byte_spf[inst.warp_id()] = 0;
    if (get_sid() == SID)
      printf("Warp %u finishes\n", inst.warp_id());
    if (m_alloc_fail_record[inst.warp_id()] != 0) {
      printf("alloc_record incorrect\n");
      printf("inst pc 0x%llx on warp %u and SM %u\n", inst.pc, inst.warp_id(), get_sid());
    }
    assert(m_alloc_fail_record[inst.warp_id()] == 0); // has to be 0 when warp is done

    m_freelist.clear();

    for (unsigned i = 0; i < m_warp_stall.size(); i++) {
      if (m_warp_stall[i]) {
        if (get_sid() == SID) {
          printf("warp %u releases\n", i);
        }
        m_warp_stall[i] = false;
        break;
      }
    }

    m_window_per_warp[inst.warp_id()].clear();

    fflush(stdout); 
  }
}

void trace_shader_core_ctx::issue_warp(register_set &warp,
                                       const warp_inst_t *pI,
                                       const active_mask_t &active_mask,
                                       unsigned warp_id, unsigned sch_id) {
  shader_core_ctx::issue_warp(warp, pI, active_mask, warp_id, sch_id);

  // delete warp_inst_t class here, it is not required anymore by gpgpu-sim
  // after issue
  // if (pI->mem_local_reg) {
  //   for (unsigned i = 0; i < pI->incount; i++) {
  //     if (pI->in[i] > 255) {
  //       printf("Delete spf instr R%u for warp %u SM %u\n", 
  //               pI->in[i], warp_id, get_sid());
  //     }
  //   }
  //   for (unsigned i = 0; i < pI->outcount; i++) {
  //     if (pI->out[i] > 255) {
  //       printf("Delete spf instr R%u for warp %u SM %u\n", 
  //               pI->out[i], warp_id, get_sid());
  //     }
  //   }
  // }
  delete pI;
}

int align_to_chunk(int number, int chunk_size) {
  if (number == 0)
    return 0;
  int num = chunk_size;
  while (num < number) {
    num += chunk_size;
  }
  return num;
}

void trace_shader_core_ctx::window_kickoff(const warp_inst_t *next_inst, unsigned warp_id) {
  // trace_shd_warp_t *m_trace_warp =
  //   static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  // trace_kernel_info_t *kernel_info = static_cast<trace_kernel_info_t *>(m_trace_warp->get_kernel_info());

  // if (kernel_info->m_tconfig->reg_win_mode == 9) {
  //   // if (get_sid() == 3 && warp_id == 39) {
  //   //   printf("regs left for warp %u on shader %u: %u\n", warp_id, get_sid(), 
  //   //           m_free_reg_per_warp[warp_id]);
  //   // }
  //   if (m_window_per_warp[warp_id].size() > 0 && 
  //         !m_window_per_warp[warp_id].back().in_reg) {
  //     // Bring own back
  //     for (unsigned i = 0; i < m_window_per_warp[warp_id].back().ldl.size(); i++) {
  //       if (m_window_per_warp[warp_id].back().ldl[i] == NULL) {
  //         printf("inst is null when bringing own back warp %u, shader %u\n", 
  //                 warp_id, get_sid());
  //         fflush(stdout);
  //       }
  //       m_spf_to_execute[warp_id].push_back(m_window_per_warp[warp_id].back().ldl[i]);
  //     }
  //     m_window_per_warp[warp_id].back().in_reg = true;

  //     // kick others out
  //     if (m_window_per_warp[warp_id].back().borrowed != -1){
  //       printf("%u is borrowed by %u on shader %u\n", warp_id, 
  //               m_window_per_warp[warp_id].back().borrowed, get_sid());
  //       fflush(stdout);
  //       assert(m_window_per_warp[warp_id].back().borrowed != -1);
  //       unsigned borrowed_from;
  //       if (warp_id >= 1) {
  //         borrowed_from = warp_id - 1;
  //       }
  //       else {
  //         borrowed_from = kernel_max_cta_per_shader * num_warps_per_cta - 1;
  //       }
  //       for (unsigned j = 0; j < 
  //             m_window_per_warp[borrowed_from][m_window_per_warp[warp_id].back().borrowed].stl.size();
  //             j++) {
  //         if (m_window_per_warp[borrowed_from][m_window_per_warp[warp_id].back().borrowed].stl[j] == NULL) {
  //           printf("inst is null when kicking others out warp %u, shader %u\n", 
  //                   warp_id, get_sid());
  //           fflush(stdout);
  //         }
  //         m_spf_to_execute[warp_id].push_back(
  //                 m_window_per_warp[borrowed_from][m_window_per_warp[warp_id].back().borrowed].stl[j]);
  //       }
  //       m_window_per_warp[borrowed_from][m_window_per_warp[warp_id].back().borrowed].in_reg = false;
  //     }
  //   }
  // }
}

// std::string to_hex_string(uint64_t num) {
//   std::ostringstream ss;
//   ss << std::hex << num;

//   return ss.str();
// }

bool trace_shader_core_ctx::has_register_space(const warp_inst_t *next_inst, unsigned warp_id, unsigned long long curr_cycle) {
  // Choose appwin or regwin?
  const trace_warp_inst_t *pI = static_cast<const trace_warp_inst_t *>(next_inst);
  trace_shd_warp_t *m_trace_warp =
    static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  trace_kernel_info_t *kernel_info = static_cast<trace_kernel_info_t *>(m_trace_warp->get_kernel_info());
  unsigned int reg_win = 0;
  unsigned int output_reg = 16;

  if (kernel_info->m_tconfig->reg_win_mode == 1) {
    reg_win = kernel_info->m_appwin + output_reg;
  } else if (kernel_info->m_tconfig->reg_win_mode == 2) {
    reg_win = kernel_info->m_kerwin + output_reg;
  } else if (kernel_info->m_tconfig->reg_win_mode == 3) {
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = align_to_chunk(pI->m_funwin, 4) + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = align_to_chunk(pI->m_funwin, 4) + output_reg;
    } else {
      return true;
    }
  } else if (kernel_info->m_tconfig->reg_win_mode == 4) {
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = align_to_chunk(pI->m_funwin, 4) + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = align_to_chunk(pI->m_funwin, 4) + output_reg;
    } else {
      return true;
    }
  } 
  else if (kernel_info->m_tconfig->reg_win_mode == 5) { // Ni: static per-warp window allocation
    // Why align_to_chunk?
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = pI->m_funwin + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = pI->m_funwin + output_reg;
    } else {
      return true;
    }
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 6) { // Ni: Ideal allocator
    // Why align_to_chunk?
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = pI->m_funwin + output_reg;
      // reg_win = kernel_info->m_max_win_single + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = pI->m_funwin + output_reg;
      // reg_win = kernel_info->m_max_win_single + output_reg;
    } else {
      return true;
    }
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 7) { // Ni: Lowmark with wrap-around 
                                                        // window spills
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = pI->m_funwin + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = pI->m_funwin + output_reg;
    } else {
      return true;
    }
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 8) { // Ni: Highmark
    // Why align_to_chunk?
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = pI->m_funwin + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = pI->m_funwin + output_reg;
    } else {
      return true;
    }
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 9) { // Ni: overflow/underflow
    // Why align_to_chunk?
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = pI->m_funwin + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = pI->m_funwin + output_reg;
    } else {
      return true;
    }
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 10) { // Ni: static w/ no concurrency limit
    // Why align_to_chunk?
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      reg_win = pI->m_funwin + output_reg;
    } else if (pI->m_opcode == OP_RET) {
      reg_win = pI->m_funwin + output_reg;
    } else {
      return true;
    }
  }
  else {
    return true;
  }

  if (kernel_info->m_tconfig->reg_win_mode == 4) {
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      // printf("%u not LIMITED by %u due to %u on SM %u\n", warp_id, m_free_reg_number, pI->m_depwin, get_sid());

      // Ni: even if the register is not enough, if the pc shows up before, can update the record and issue
      bool flag = false;
      for (int i = 0; i < m_call_record[warp_id].size(); i++) {
        if (pI->a_pc == m_call_record[warp_id][i].first) {
          flag = true;
          if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
            printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
                    pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
            fflush(stdout);
          }
          assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
          m_call_record[warp_id][i].second |= pI->get_active_mask();
          break;
        }
      }
      if (!flag) {
        m_dep_table[warp_id][0] = 1;
        m_dep_table[warp_id][1] = pI->m_depwin;
        m_dep_table[warp_id][3] = pI->a_pc;
        int sum = 0;
        for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
          m_dep_table[k][2] = m_warp[k]->waiting() ? 0 : 1;
        }
        for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
          // sum += m_dep_table[k][0] * align_to_chunk(m_dep_table[k][1], 32) * m_dep_table[warp_id][2];
          // Ni: other hs + own ws
          if (k == warp_id) {
            // sum += m_dep_table[k][0] * reg_win * m_dep_table[warp_id][2];
            sum += m_dep_table[k][0] * align_to_chunk(reg_win, 32) * m_dep_table[warp_id][2];
          }
          else {
            // sum += m_dep_table[k][0] * m_dep_table[k][1] * m_dep_table[warp_id][2];
            sum += m_dep_table[k][0] * align_to_chunk(m_dep_table[k][1], 32) * m_dep_table[warp_id][2];
          }
        }

        if (sum <= m_free_reg_number) {
          active_mask_t mask_temp(pI->get_active_mask());
          m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));
          m_alloc_fail_record[warp_id]++;
          m_free_reg_number -= reg_win;
          m_dep_table[warp_id][1] -= reg_win;

          if (get_sid() == SID) {
            printf("Reserve 0x%llx free reg left for SM %u warp %u: %u\n", next_inst->a_pc, get_sid(), warp_id,  m_free_reg_number);
          }

          if (get_sid() == SID && warp_id == WID) {
            // printf("0x%llx free reg left for SM %u: %u\n", next_inst->a_pc, get_sid(), m_free_reg_number);
            
            for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
              printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                      m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
            }
          }
          return true;
        }
        else {
          m_dep_table[warp_id][0] = 0;
          m_dep_table[warp_id][1] = 0;
          // if (get_sid() == SID) {
          //   printf("RCWS return false 0x%llx on SM %u warp %u\n", pI->a_pc, get_sid(), warp_id);
          // }
          if (get_sid() == SID && warp_id == WID) {
            printf("alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
            printf("RCWS return false 0x%llx on SM %u warp %u\n", pI->a_pc, get_sid(), warp_id);
            printf("sum: %d, remain: %d\n", sum, m_free_reg_number);
            printf("Due to warps: ");
            for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
              if (m_dep_table[k][0] * m_dep_table[k][1] * m_dep_table[warp_id][2] != 0) {
                printf("%u: %d 0x%llx now at 0x%llx, ", k, m_dep_table[warp_id][1], m_dep_table[warp_id][3], m_warp[warp_id]->get_pc());
              }
            }
            printf("\n");

            for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
              printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                      m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
            }
            fflush(stdout);
          }
          return false;
        }
      }
      else {
        return true;
      }
    } else if (pI->m_opcode == OP_RET){
      if (get_sid() == SID && warp_id == WID)
        printf("RET\n");
      // m_free_reg_number += reg_win;
      active_mask_t mask_temp = pI->get_active_mask();
      if (get_sid() == SID && warp_id == WID)
        printf("pI mask %llx, record mask %llx\n", mask_temp.to_ullong(),
                m_call_record[warp_id].back().second);
      fflush(stdout);
      if (mask_temp.any()) {
        for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
          bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
          if (ret_true) {
            for (unsigned k = 0; k < WARP_SIZE; k++) {
              if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                mask_temp.reset(k);
                m_call_record[warp_id][j].second.reset(k);
              }
            }
            if (mask_temp.none()) {
              break;
            }
          }
        }
      }

      while (m_call_record[warp_id].back().second == 0) {
        if (get_sid() == SID && warp_id == WID)
          printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
        m_call_record[warp_id].pop_back();
        m_free_reg_number += reg_win;
        assert(m_alloc_fail_record[warp_id] >= 1);
        m_alloc_fail_record[warp_id]--;
        if (get_sid() == SID) {
          printf("Release 0x%llx free reg left for SM %u warp %u: %u\n", next_inst->a_pc, get_sid(), warp_id,  m_free_reg_number);
        }
      }
      if (m_alloc_fail_record[warp_id] == 0) {
        m_dep_table[warp_id][1] = 0;
      }
      fflush(stdout);
      // printf("Reg return %d %d!\n", m_free_reg_number, reg_win);
      // printf("Release %u number of registers at cycle %llu on SM %u\n", reg_win, curr_cycle, get_sid());
      return true;
    }
    // Will never calls this
    return false;
  } 
  else if (kernel_info->m_tconfig->reg_win_mode == 5) {
    if (!m_start_ker[warp_id]) {
      if (m_free_reg_per_warp[warp_id] >= (kernel_info->m_ker_local_win + 32)) {
        m_free_reg_per_warp[warp_id] -= (kernel_info->m_ker_local_win + 32);
      }
      else {
        m_ker_fail[warp_id] = true;
      }
      if (warp_id == WID) {
        printf("m_free_reg_per_warp now is %d\n", m_free_reg_per_warp[warp_id]);
        fflush(stdout);
      }
      m_start_ker[warp_id] = true;
    }
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      bool flag = false;
      for (int i = 0; i < m_call_record[warp_id].size(); i++) {
        if (pI->a_pc == m_call_record[warp_id][i].first) {
          flag = true;
          if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
            printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
                    pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
            fflush(stdout);
          }
          assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
          m_call_record[warp_id][i].second |= pI->get_active_mask();
          break;
        }
      }
      if (!flag) {
        total_func_call[warp_id]++;
        active_mask_t mask_temp(pI->get_active_mask());
        m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));
        if (m_free_reg_per_warp[warp_id] >= reg_win && !m_ker_fail[warp_id]) {
          m_free_reg_per_warp[warp_id] -= reg_win;
          if (get_sid() == SID && warp_id == WID)
            printf("0x%llx free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
        }
        else {
          func_call_spf[warp_id]++;
          m_alloc_fail_record[warp_id]++;
          printf("func needs %d, but have %d left\n", reg_win, m_free_reg_per_warp[warp_id]);
          if (get_sid() == SID && warp_id == WID)
            printf("alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
        }
      }
      if (get_sid() == SID && warp_id == WID) {
        for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
          printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                  m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
        }
      }
      
      // fflush(stdout);
      return true; // Ni: start spills&fills after regs exhausted
    }
    else if (pI->m_opcode == OP_RET) {
      bool flag = true;
      active_mask_t mask_temp = pI->get_active_mask();
      // if (get_sid() == SID && warp_id == WID)
      //     printf("pI mask %llx, record mask %llx\n", mask_temp.to_ullong(),
      //             m_call_record[warp_id].back().second);
      if (mask_temp.any()) {
        for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
          bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
          if (ret_true) {
            std::string mask1 = m_call_record[warp_id][j].second.to_string();
            std::string mask2 = mask_temp.to_string();
            if (mask1 >= mask2) {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
              // m_call_record[warp_id][j].second = m_call_record[warp_id][j].second ^ mask_temp; // bug
              // mask_temp.reset();
            }
            else {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
            }
            if (mask_temp.none()) {
              break;
            }
          }
        }
      }

      while (m_call_record[warp_id].back().second == 0) {
      // if (m_call_record[warp_id].back().second == 0) {
        if (get_sid() == SID && warp_id == WID)
          printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
        m_call_record[warp_id].pop_back();
        if (m_alloc_fail_record[warp_id] > 0) {
          m_alloc_fail_record[warp_id]--;
          if (get_sid() == SID && warp_id == WID)
            printf("RET alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
        }
        else if (m_alloc_fail_record[warp_id] == 0) {
          m_free_reg_per_warp[warp_id] += reg_win;
          if (get_sid() == SID && warp_id == WID)
            printf("0x%llx RET free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
        }
      }
      fflush(stdout);
      return true;
    }
    else {
      return true;
    }
    
    return true; 
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 6) {
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      if (!m_start_ker[warp_id]) {
        m_free_reg_number -= (kernel_info->m_ker_local_win + 32);
        if (warp_id == WID) {
          printf("m_free_reg_number now is %d\n", m_free_reg_number);
          fflush(stdout);
        }
        m_start_ker[warp_id] = true;
      }
      bool flag = false;
      for (int i = 0; i < m_call_record[warp_id].size(); i++) {
        if (pI->a_pc == m_call_record[warp_id][i].first) {
          flag = true;
          if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
            printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
                    pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
            fflush(stdout);
          }
          assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
          m_call_record[warp_id][i].second |= pI->get_active_mask();
          break;
        }
      }
      if (!flag) {
        if (SID == get_sid())
          printf("Reserve %u number of registers at cycle %llu on SM %u\n", reg_win, curr_cycle, get_sid());
        total_func_call[warp_id]++;
        active_mask_t mask_temp(pI->get_active_mask());
        m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));

        if (m_free_reg_number >= reg_win) {
          m_free_reg_number -= reg_win;
          m_pair_record[warp_id].push_back(false);
          if (get_sid() == SID && warp_id == WID)
            printf("0x%llx free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
        }
        else {
          func_call_spf[warp_id]++;
          m_alloc_fail_record[warp_id]++;
          m_pair_record[warp_id].push_back(true);
          if (get_sid() == SID && warp_id == WID)
            printf("alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
        }
      }
      if (get_sid() == SID && warp_id == WID) {
        for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
          printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                  m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
        }
      }
      
      // fflush(stdout);
      return true; // Ni: start spills&fills after regs exhausted
    }
    else if (pI->m_opcode == OP_RET) {
      bool flag = true;
      active_mask_t mask_temp = pI->get_active_mask();
      // if (get_sid() == SID && warp_id == WID)
      //     printf("pI mask %llx, record mask %llx\n", mask_temp.to_ullong(),
      //             m_call_record[warp_id].back().second);
      if (mask_temp.any()) {
        for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
          bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
          if (ret_true) {
            std::string mask1 = m_call_record[warp_id][j].second.to_string();
            std::string mask2 = mask_temp.to_string();
            if (mask1 >= mask2) {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
              // m_call_record[warp_id][j].second = m_call_record[warp_id][j].second ^ mask_temp; // bug
              // mask_temp.reset();
            }
            else {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
            }
            if (mask_temp.none()) {
              break;
            }
          }
        }
      }

      while (m_call_record[warp_id].back().second == 0) {
        if (SID == get_sid())
          printf("Release %u number of registers at cycle %llu on SM %u\n", reg_win, curr_cycle, get_sid());
        if (get_sid() == SID && warp_id == WID)
          printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
        m_call_record[warp_id].pop_back();
        m_pair_record[warp_id].pop_back();
        if (m_alloc_fail_record[warp_id] > 0) {
          m_alloc_fail_record[warp_id]--;
          if (get_sid() == SID && warp_id == WID)
            printf("RET alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
        }
        else if (m_alloc_fail_record[warp_id] == 0) {
          m_free_reg_number += reg_win;
          if (get_sid() == SID && warp_id == WID)
            printf("0x%llx RET free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
        }
      }
      fflush(stdout);
      return true;
    }
    else {
      return true;
    }
    
    return true; 
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 7) {
    if (!m_start_ker[warp_id]) {
      if (m_free_reg_per_warp[warp_id] >= (kernel_info->m_ker_local_win + 32)) {
        m_free_reg_per_warp[warp_id] -= (kernel_info->m_ker_local_win + 32);
        std::pair<unsigned, bool> in = std::make_pair(16, true);
        std::pair<unsigned, bool> local = std::make_pair(kernel_info->m_ker_local_win, true);
        std::pair<unsigned, bool> out = std::make_pair(16, true);
        std::vector<std::pair<unsigned, bool>> func;
        func.push_back(in);
        func.push_back(local);
        func.push_back(out);
        m_win_record[warp_id].push_back(func);
      }
      else {
        printf("ERROR: Not enough registers for the base kernel\n");
        fflush(stdout);
        abort();
      }
      if (warp_id == WID) {
        printf("m_free_reg_per_warp now is %d\n", m_free_reg_per_warp[warp_id]);
        fflush(stdout);
      }
      m_start_ker[warp_id] = true;
    }
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      bool flag = false;
      for (int i = 0; i < m_call_record[warp_id].size(); i++) {
        if (pI->a_pc == m_call_record[warp_id][i].first) {
          flag = true;
          if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
            printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
                    pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
            fflush(stdout);
          }
          assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
          m_call_record[warp_id][i].second |= pI->get_active_mask();
          break;
        }
      }
      if (!flag) {
        total_func_call[warp_id]++;
        active_mask_t mask_temp(pI->get_active_mask());
        m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));
        if (m_free_reg_per_warp[warp_id] >= reg_win) {
          m_free_reg_per_warp[warp_id] -= reg_win;

          std::pair<unsigned, bool> local = std::make_pair(reg_win-16, true);
          std::pair<unsigned, bool> out = std::make_pair(16, true);
          std::vector<std::pair<unsigned, bool>> func;
          func.push_back(local);
          func.push_back(out);
          m_win_record[warp_id].push_back(func);

          if (get_sid() == SID && warp_id == WID)
            printf("0x%llx free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
        }
        else {
          func_call_spf[warp_id]++;
          // TODO: push window spill instructions
          // Need to update m_free_reg_per_warp too

          // 1. Use remaining free regs (update free_reg)
          if (m_free_reg_per_warp[warp_id] != 0) {
            assert(reg_win > m_free_reg_per_warp[warp_id]);
            reg_win -= m_free_reg_per_warp[warp_id];
            m_free_reg_per_warp[warp_id] = 0;
          }

          // 2. spill windows, update record and update the free_reg 
          //    (spill from the beginning of the vec)
          assert(reg_win > 0);
          bool enough = false;
          for (unsigned m = 0; m < m_win_record[warp_id].size(); m++) {
            for (unsigned n = 0; n < m_win_record[warp_id][m].size(); n++) {
              if (m_win_record[warp_id][m][n].second) {
                for (unsigned spf = 0; spf < m_win_record[warp_id][m][n].first; spf++) {
                  // inst_trace_t* spf_inst = new inst_trace_t();
                  // std::string* stl_line = new std::string(" ");
                  // *stl_line = to_hex_string(apc_start) + " " + to_hex_string(pc_start) + 
                  //                   " " + to_hex_string(mask_temp.to_ulong()) + 
                  //                   " 0 STL 2 R1 R" + std::to_string(reg_next) + " 1 4 1 0x" + 
                  //                   to_hex_string(stack_start) + " 0 ";
                  // char buf[100];
                  char* buf = new char[100];
                  sprintf(buf, "%x %x %x 0 STL 2 R1 R%d 1 4 1 0x%x 0 ", apc_start, pc_start, 
                                mask_temp, reg_next, stack_start);
                  if (warp_id == WID && get_sid() == SID) {
                    // printf("%s\n", buf);
                    std::cout << buf;
                  }
                  // std::string* stl_line = new std::string(buf);
                  
                  // assert(spf_inst->parse_from_string(stl_line, 3));
                  m_spf_to_execute[warp_id].push(buf);
                  // test
                  // delete stl_line;

                  // std::string ldl_line = to_hex_string(apc_start) + " " + to_hex_string(pc_start) + 
                  //                   to_hex_string(mask_temp.to_ulong()) + 
                  //                   " 1 R" + std::to_string(reg_next) + " LDL 1 R1 1 4 1 0x" + 
                  //                   to_hex_string(stack_start) + " 0 ";
                  // std::string* ldl_line = new std::string();
                  // *ldl_line = to_hex_string(apc_start) + " " + to_hex_string(pc_start) + 
                  //                   to_hex_string(mask_temp.to_ulong()) + 
                  //                   " 1 R" + std::to_string(reg_next) + " LDL 1 R1 1 4 1 0x" + 
                  //                   to_hex_string(stack_start) + " 0 ";
                  if (spf == (m_win_record[warp_id][m][n].first - 1)) {
                    // std::vector<std::string*> sub_win;
                    // sub_win.push_back(ldl_line);
                    m_reg_stack_map[warp_id].push_back(std::make_pair(stack_start, mask_temp.to_ulong()));
                  }
                  // else {
                  //   m_reg_stack_map[warp_id].back().push_back(ldl_line);
                  // }
                  apc_start += 0x10;
                  pc_start += 0x10;
                  reg_next++;
                  stack_start -= 0x4;
                }
                m_win_record[warp_id][m][n].first = false;
                // assert(reg_next < 255);

                if (reg_win <= m_win_record[warp_id][m][n].first) {
                  m_free_reg_per_warp[warp_id] += (m_win_record[warp_id][m][n].first - reg_win);
                  reg_win = 0;
                  enough = true;
                  break;
                }
                else {
                  reg_win -= m_win_record[warp_id][m][n].first;
                }
              }
            }
            if (enough) {
              break;
            }
          }
          apc_start = 0x0;
          pc_start = 0x0;
          reg_next = 256;
          
          std::pair<unsigned, bool> local = std::make_pair(reg_win-16, true);
          std::pair<unsigned, bool> out = std::make_pair(16, true);
          std::vector<std::pair<unsigned, bool>> func;
          func.push_back(local);
          func.push_back(out);
          m_win_record[warp_id].push_back(func);

          // printf("func needs %d, but have %d left\n", reg_win, m_free_reg_per_warp[warp_id]);
          // if (get_sid() == SID && warp_id == WID)
          //   printf("alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
        }
      }
      if (get_sid() == SID && warp_id == WID) {
        for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
          printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                  m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
        }
      }
      
      // fflush(stdout);
      return true;
    }
    else if (pI->m_opcode == OP_RET) {
      bool flag = true;
      active_mask_t mask_temp = pI->get_active_mask();
      if (mask_temp.any()) {
        for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
          bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
          if (ret_true) {
            std::string mask1 = m_call_record[warp_id][j].second.to_string();
            std::string mask2 = mask_temp.to_string();
            if (mask1 >= mask2) {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
            }
            else {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
            }
            if (mask_temp.none()) {
              break;
            }
          }
        }
      }

      while (m_call_record[warp_id].back().second == 0) {
        if (get_sid() == SID && warp_id == WID)
          printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
        m_call_record[warp_id].pop_back();

        // Pop the current windows and free registers
        for (unsigned m = 0; m < m_win_record[warp_id].back().size(); m++) {
          m_free_reg_per_warp[warp_id] += m_win_record[warp_id].back()[m].first;
          assert(m_win_record[warp_id].back()[m].second);
        }
        m_win_record[warp_id].pop_back();

        // TODO: fill the window back
        // 1. Fill the windows, including the previous output reg if not base (back-1)

        // 2. update the free_reg (-)
        for (int m = m_win_record[warp_id].back().size()-1; m >= 0; m--) {
          if (!m_win_record[warp_id].back()[m].second) {
            // TODO: fill windows
            // if (m_win_record[warp_id].back()[m].first != m_reg_stack_map[warp_id].back().size()) {
            //   printf("warp %u SM %u\n", warp_id, get_sid());
            //   for (unsigned func = 0; func < m_win_record[warp_id].size(); func++) {
            //     printf("func %u\n", func);
            //     for (unsigned subwin = 0; subwin < m_win_record[warp_id][func].size(); subwin++) {
            //       printf("%u %u %u\n", subwin, 
            //               m_win_record[warp_id][func][subwin].first, 
            //               m_win_record[warp_id][func][subwin].second);
            //     }
            //   }
            //   printf("ldl instrs\n");
            //   for (unsigned subwin = 0; subwin < m_reg_stack_map[warp_id].size(); subwin++) {
            //     printf("%u\n", m_reg_stack_map[warp_id][subwin].size());
            //   }
            //   fflush(stdout);
            // }
            // assert(m_win_record[warp_id].back()[m].first == m_reg_stack_map[warp_id].back().size());
            stack_start = m_reg_stack_map[warp_id].back().first;
            for (int spf = m_win_record[warp_id].back()[m].first; spf > 0; spf--) {
              // inst_trace_t* spf_inst = new inst_trace_t();
              // std::string* ldl_line = m_reg_stack_map[warp_id].back().back();
              // std::string* ldl_line = new std::string(" ");
              // *ldl_line = to_hex_string(apc_start) + " " + to_hex_string(pc_start) + 
              //                       to_hex_string(m_reg_stack_map[warp_id].back().second) + 
              //                       " 1 R" + std::to_string(reg_next) + " LDL 1 R1 1 4 1 0x" + 
              //                       to_hex_string(stack_start) + " 0 ";
              // char buf[100];
              char* buf = new char[100];
              sprintf(buf, "%x %x %x 1 R%d LDL 1 R1 1 4 1 0x%x 0 ", apc_start, pc_start, 
                            m_reg_stack_map[warp_id].back().second, reg_next, stack_start);
              // std::string* ldl_line = new std::string(buf);
              // assert(spf_inst->parse_from_string(ldl_line, 3));

              // std::vector<std::pair<unsigned, unsigned>>::iterator ldl_delete = 
              //                                 m_reg_stack_map[warp_id].end();
              // delete *ldl_delete;
              // m_reg_stack_map[warp_id].erase(ldl_delete);
              // trace_warp_inst_t *new_inst =
              //       new trace_warp_inst_t(m_config);
              // new_inst->parse_from_trace_struct(
              //     *spf_inst, kernel_info->OpcodeMap,
              //     kernel_info->m_tconfig, kernel_info->m_kernel_trace_info);
              // delete spf_inst;
              m_spf_to_execute[warp_id].push(buf);
              // test
              // delete ldl_line;

              apc_start += 0x10;
              pc_start += 0x10;
              reg_next++;
              stack_start += 0x4;
            }
            // assert(m_reg_stack_map[warp_id].back().size() == 0);
            std::vector<std::pair<unsigned, unsigned long>>::iterator subwin_delete = 
                                              m_reg_stack_map[warp_id].end();
            m_reg_stack_map[warp_id].erase(subwin_delete);
            m_win_record[warp_id].back()[m].second = true;
            assert(m_free_reg_per_warp[warp_id] >= m_win_record[warp_id].back()[m].first);
            m_free_reg_per_warp[warp_id] -= m_win_record[warp_id].back()[m].first;
          }
        }

        // fill previous output reg if not base
        unsigned num_funcs = m_win_record[warp_id].size();
        if (num_funcs > 1) {
          if (!m_win_record[warp_id][num_funcs-2].back().second) {
            // TODO fill the output window
            // assert(m_win_record[warp_id][num_funcs-2].back().first == m_reg_stack_map[warp_id].back().size());
            stack_start = m_reg_stack_map[warp_id].back().first;
            for (int spf = m_win_record[warp_id][num_funcs-2].back().first; spf = 0; spf--) {
              // inst_trace_t* spf_inst;
              // std::string* ldl_line = new std::string(" ");
              // *ldl_line = to_hex_string(apc_start) + " " + to_hex_string(pc_start) + 
              //                       to_hex_string(m_reg_stack_map[warp_id].back().second) + 
              //                       " 1 R" + std::to_string(reg_next) + " LDL 1 R1 1 4 1 0x" + 
              //                       to_hex_string(stack_start) + " 0 ";
              // char buf[100];
              char* buf = new char[100];
              sprintf(buf, "%x %x %x 1 R%d LDL 1 R1 1 4 1 0x%x 0 ", apc_start, pc_start, 
                            m_reg_stack_map[warp_id].back().second, reg_next, stack_start);
              // std::string* ldl_line = new std::string(buf);
              // assert(spf_inst->parse_from_string(ldl_line, 3));
              // std::vector<std::pair<unsigned, unsigned>>::iterator ldl_delete = 
              //                                 m_reg_stack_map[warp_id].end();
              // delete *ldl_delete;
              // m_reg_stack_map[warp_id].erase(ldl_delete);
              // trace_warp_inst_t *new_inst =
              //       new trace_warp_inst_t(m_config);
              // new_inst->parse_from_trace_struct(
              //     *spf_inst, kernel_info->OpcodeMap,
              //     kernel_info->m_tconfig, kernel_info->m_kernel_trace_info);
              // delete spf_inst;
              m_spf_to_execute[warp_id].push(buf);
              // test
              // delete ldl_line;

              apc_start += 0x10;
              pc_start += 0x10;
              reg_next++;
              stack_start += 0x4;
            }
            std::vector<std::pair<unsigned, unsigned long>>::iterator subwin_delete = 
                                              m_reg_stack_map[warp_id].end();
            m_reg_stack_map[warp_id].erase(subwin_delete);
            m_win_record[warp_id][num_funcs-2].back().second = true;
            assert(m_free_reg_per_warp[warp_id] >= m_win_record[warp_id][num_funcs-2].back().first);
            m_free_reg_per_warp[warp_id] -= m_win_record[warp_id][num_funcs-2].back().first;
          }
        }

        apc_start = 0x0;
        pc_start = 0x0;
        reg_next = 256;
      }
      fflush(stdout);
      return true;
    }
    else {
      return true;
    }
    
    return true; 
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 8) {
    if (m_warp_stall[warp_id]) {
      // bool release_flag = false;
      // unsigned cta_in_shader = warp_id / num_warps_per_cta;
      // for (unsigned i = 0; i < num_warps_per_cta; i++) {
      //   if (m_warp[cta_in_shader*num_warps_per_cta+i]->waiting()) {
      //     release_flag = true;  // release but do spill&fill
      //     break;
      //   }
      // }
      // if (!release_flag) {
      if (get_sid() == SID) {
        printf("warp %u is stuck\n", warp_id);
        fflush(stdout);
      }
        return false; // The warp is stalled
      // }
    }
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      bool flag = false;
      for (int i = 0; i < m_call_record[warp_id].size(); i++) {
        if (pI->a_pc == m_call_record[warp_id][i].first) {
          flag = true;
          if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
            printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
                    pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
            fflush(stdout);
          }
          assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
          m_call_record[warp_id][i].second |= pI->get_active_mask();
          break;
        }
      }
      if (!flag) {
        total_func_call[warp_id]++;
        active_mask_t mask_temp(pI->get_active_mask());
        m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));
        if (m_warp_stall[warp_id]) {
          func_call_spf[warp_id]++;
          m_pair_record[warp_id].push_back(true);
          abort();
        }
        else {
          if (stall_all) {
            if (reg_win <= m_free_reg_per_warp[warp_id]) {
              m_free_reg_per_warp[warp_id] -= reg_win;
              m_pair_record[warp_id].push_back(false);
            }
            else {
              m_pair_record[warp_id].push_back(true);
            }
            abort();
          }
          else {
            m_pair_record[warp_id].push_back(false);
          }
        }
      }
      if (get_sid() == SID && warp_id == WID) {
        for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
          printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                  m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
        }
      }
      
      // fflush(stdout);
      return true; // Ni: start spills&fills after regs exhausted
    }
    else if (pI->m_opcode == OP_RET) {
      bool flag = true;
      active_mask_t mask_temp = pI->get_active_mask();
      // if (get_sid() == SID && warp_id == WID)
      //     printf("pI mask %llx, record mask %llx\n", mask_temp.to_ullong(),
      //             m_call_record[warp_id].back().second);
      if (mask_temp.any()) {
        for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
          bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
          if (ret_true) {
            std::string mask1 = m_call_record[warp_id][j].second.to_string();
            std::string mask2 = mask_temp.to_string();
            if (mask1 >= mask2) {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
              // m_call_record[warp_id][j].second = m_call_record[warp_id][j].second ^ mask_temp; // bug
              // mask_temp.reset();
            }
            else {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
            }
            if (mask_temp.none()) {
              break;
            }
          }
        }
      }

      while (m_call_record[warp_id].back().second == 0) {
        if (get_sid() == SID && warp_id == WID)
          printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
        if (stall_all) {
          m_free_reg_per_warp[warp_id] += reg_win;
          abort();
        }
        m_call_record[warp_id].pop_back();
        m_pair_record[warp_id].pop_back();
      }
      fflush(stdout);
      return true;
    }
    else {
      return true;
    }
    
    return true; 
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 9) {
    // if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
    //   bool flag = false;
    //   for (int i = 0; i < m_call_record[warp_id].size(); i++) {
    //     if (pI->a_pc == m_call_record[warp_id][i].first) {
    //       flag = true;
    //       if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
    //         printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
    //                 pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
    //         fflush(stdout);
    //       }
    //       assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
    //       m_call_record[warp_id][i].second |= pI->get_active_mask();
    //       break;
    //     }
    //   }
    //   if (!flag) {
    //     total_func_call[warp_id]++;
    //     active_mask_t mask_temp(pI->get_active_mask());
    //     m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));
    //     m_window_per_warp[warp_id].back().reg_num = reg_win;
    //     m_window_per_warp[warp_id].back().in_reg = true;
    //     m_window_per_warp[warp_id].back().borrowed = -1;
    //     if (m_free_reg_per_warp[warp_id] >= reg_win) {
    //       m_window_per_warp[warp_id].back().used_own = reg_win;
    //       m_free_reg_per_warp[warp_id] -= reg_win;
    //       if (get_sid() == SID && warp_id == WID)
    //         printf("0x%llx used own: %u\n", pI->a_pc, reg_win);
    //     }
    //     else {
    //       func_call_spf[warp_id]++;
    //       int reg_used = reg_win;
    //       unsigned total_next_window_avail = 0;
    //       unsigned borrow_from = (warp_id + 1) % (kernel_max_cta_per_shader * num_warps_per_cta);
    //       total_next_window_avail += m_free_reg_per_warp[warp_id];
    //       for (unsigned i = 0; i < m_window_per_warp[borrow_from].size(); i++) {
    //         if (m_window_per_warp[borrow_from][i].borrowed == -1) {
    //           total_next_window_avail += m_window_per_warp[borrow_from][i].reg_num;
    //           if (get_sid() == SID && warp_id == WID)
    //             printf("0x%llx borrowed from warp %u: %u\n", pI->a_pc, borrow_from, 
    //                       m_window_per_warp[borrow_from][i].reg_num);
    //         }
    //         else {
    //           printf("shader %u warp %u index %u reg_num %u borrowed by %u\n", 
    //                   get_sid(), borrow_from, i, m_window_per_warp[borrow_from][i].reg_num, 
    //                   m_window_per_warp[borrow_from][i].borrowed);
    //           // assert(false);
    //         }
    //       }
    //       if (reg_used > total_next_window_avail) {
    //         total_next_window_avail += m_free_reg_per_warp[borrow_from];
    //         printf("own left: %u\n", m_free_reg_per_warp[warp_id]);
    //         printf("others left: %u on warp %u\n", m_free_reg_per_warp[borrow_from], borrow_from);
    //         printf("reg_used: %u, total_next_window_avail: %u, in warp %u, shader %u\n", 
    //                 reg_used, total_next_window_avail, warp_id, get_sid());
    //         fflush(stdout);
    //       }
    //       assert(reg_used <= total_next_window_avail);
    //       m_window_per_warp[warp_id].back().used_own = m_free_reg_per_warp[warp_id];
    //       reg_used -= m_free_reg_per_warp[warp_id];
    //       m_free_reg_per_warp[warp_id] = 0;
          
    //       for (unsigned i = 0; i < m_window_per_warp[borrow_from].size(); i++) {
    //         if (m_window_per_warp[borrow_from][i].in_reg) {
    //           reg_used -= m_window_per_warp[borrow_from][i].reg_num;
    //           m_window_per_warp[borrow_from][i].in_reg = false;
    //           for (unsigned j = 0; j < m_window_per_warp[borrow_from][i].stl.size(); j++) {
    //             if (m_window_per_warp[borrow_from][i].stl[j] == NULL) {
    //               printf("inst is null when borrowing window warp %u, shader %u\n", 
    //                       warp_id, get_sid());
    //             }
    //             m_spf_to_execute[warp_id].push_back(m_window_per_warp[borrow_from][i].stl[j]);
    //           }
    //           m_window_per_warp[warp_id].back().reg_num = reg_win;
    //           m_window_per_warp[warp_id].back().in_reg = true;
    //           m_window_per_warp[warp_id].back().borrowed = -1;
    //           m_window_per_warp[warp_id].back().borrow.push_back(i);
    //           m_window_per_warp[borrow_from].back().borrowed = m_window_per_warp[warp_id].size()-1;
    //           // if (get_sid() == 9)
    //           //   printf("CALL %u is borrowed by %u\n", borrow_from, warp_id);
    //           fflush(stdout);
    //         }
    //         if (reg_used < 0) {
    //           break;
    //         }
    //       }

    //       // TODO: what if reg_used is still greater than 0?
    //       // if (reg_used > 0) {
    //       //   m_free_reg_per_warp[warp_id+1] -= reg_used;
    //       //   m_window_per_warp[warp_id+1].back().borrowed_size = reg_used;
    //       //   assert(m_free_reg_per_warp[warp_id+1] >= 0);
    //       // }
    //       // printf("func needs %d, but have %d left\n", reg_win, m_free_reg_per_warp[warp_id]);
    //       // if (get_sid() == SID && warp_id == WID)
    //       //   printf("alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
    //     }
    //   }
    //   if (get_sid() == SID && warp_id == WID) {
    //     for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
    //       printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
    //               m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
    //     }
    //   }
      
    //   // fflush(stdout);
    //   return true; // Ni: start spills&fills after regs exhausted
    // }
    // else if (pI->m_opcode == OP_RET) {
    //   bool flag = true;
    //   active_mask_t mask_temp = pI->get_active_mask();
    //   // if (get_sid() == SID && warp_id == WID)
    //   //     printf("pI mask %llx, record mask %llx\n", mask_temp.to_ullong(),
    //   //             m_call_record[warp_id].back().second);
    //   if (mask_temp.any()) {
    //     for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
    //       bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
    //       if (ret_true) {
    //         std::string mask1 = m_call_record[warp_id][j].second.to_string();
    //         std::string mask2 = mask_temp.to_string();
    //         if (mask1 >= mask2) {
    //           for (unsigned k = 0; k < WARP_SIZE; k++) {
    //             if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
    //               mask_temp.reset(k);
    //               m_call_record[warp_id][j].second.reset(k);
    //             }
    //           }
    //           // m_call_record[warp_id][j].second = m_call_record[warp_id][j].second ^ mask_temp; // bug
    //           // mask_temp.reset();
    //         }
    //         else {
    //           for (unsigned k = 0; k < WARP_SIZE; k++) {
    //             if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
    //               mask_temp.reset(k);
    //               m_call_record[warp_id][j].second.reset(k);
    //             }
    //           }
    //         }
    //         if (mask_temp.none()) {
    //           break;
    //         }
    //       }
    //     }
    //   }

    //   while (m_call_record[warp_id].back().second == 0) {
    //   // if (m_call_record[warp_id].back().second == 0) {
    //     if (get_sid() == SID && warp_id == WID)
    //       printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
    //     m_call_record[warp_id].pop_back();
    //     m_free_reg_per_warp[warp_id] += m_window_per_warp[warp_id].back().used_own;
    //     if (get_sid() == SID && warp_id == WID)
    //       printf("%llx pop window %u\n", pI->a_pc, m_window_per_warp[warp_id].size()-1);
    //     m_window_per_warp[warp_id].pop_back();
    //     assert(m_window_per_warp[warp_id].size() > 0);
    //     // if (m_alloc_fail_record[warp_id] > 0) {
    //     //   m_alloc_fail_record[warp_id]--;
    //     //   if (get_sid() == SID && warp_id == WID)
    //     //     printf("RET alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
    //     // }
    //     // else if (m_alloc_fail_record[warp_id] == 0) {
    //     //   m_free_reg_per_warp[warp_id] += reg_win;
    //     //   if (get_sid() == SID && warp_id == WID)
    //     //     printf("0x%llx RET free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
    //     // }
    //   }
    //   fflush(stdout);
    //   return true;
    // }
    // else {
    //   return true;
    // }
    
    // return true; 
  }
  else if (kernel_info->m_tconfig->reg_win_mode == 10) {
    if (m_warp_stall[warp_id]) {
      // bool release_flag = false;
      // unsigned cta_in_shader = warp_id / num_warps_per_cta;
      // for (unsigned i = 0; i < num_warps_per_cta; i++) {
      //   if (m_warp[cta_in_shader*num_warps_per_cta+i]->waiting()) {
      //     release_flag = true;  // release but do spill&fill
      //     break;
      //   }
      // }
      // if (!release_flag) {
      if (get_sid() == SID) {
        printf("warp %u is stuck\n", warp_id);
        fflush(stdout);
      }
        return false; // The warp is stalled
      // }
    }
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      bool flag = false;
      for (int i = 0; i < m_call_record[warp_id].size(); i++) {
        if (pI->a_pc == m_call_record[warp_id][i].first) {
          flag = true;
          if ((m_call_record[warp_id][i].second & pI->get_active_mask()) != 0) {
            printf("Wrong CALL pc: 0x%llx, mask: %llx&%llx on SM %u warp %u\n", 
                    pI->a_pc, m_call_record[warp_id][i].second, pI->get_active_mask(), get_sid(), warp_id);
            fflush(stdout);
          }
          assert((m_call_record[warp_id][i].second & pI->get_active_mask()) == 0);
          m_call_record[warp_id][i].second |= pI->get_active_mask();
          break;
        }
      }
      if (!flag) {
        total_func_call[warp_id]++;
        active_mask_t mask_temp(pI->get_active_mask());
        m_call_record[warp_id].push_back(std::make_pair(pI->a_pc, mask_temp));
        if (m_free_reg_per_warp[warp_id] >= reg_win) {
          m_free_reg_per_warp[warp_id] -= reg_win;
          m_pair_record[warp_id].push_back(false);
          if (get_sid() == SID && warp_id == WID)
            printf("0x%llx free reg left for warp %u: %u\n", next_inst->a_pc, warp_id, m_free_reg_per_warp[warp_id]);
        }
        else {
          if (m_warp_stall[warp_id]) {
            func_call_spf[warp_id]++;
            m_pair_record[warp_id].push_back(true);
            abort();
          }
          else {
            if (stall_all) {
              if (reg_win <= m_free_reg_per_warp[warp_id]) {
                m_free_reg_per_warp[warp_id] -= reg_win;
                m_pair_record[warp_id].push_back(false);
              }
              else {
                func_call_spf[warp_id]++;
                m_pair_record[warp_id].push_back(true);
              }
              abort();
            }
            else {
              func_call_spf[warp_id]++;
              m_pair_record[warp_id].push_back(true);
            }
          }
          printf("func needs %d, but have %d left\n", reg_win, m_free_reg_per_warp[warp_id]);
          if (get_sid() == SID && warp_id == WID)
            printf("alloc_record 0x%llx for warp %u: %u\n", next_inst->a_pc, warp_id, m_alloc_fail_record[warp_id]);
        }
      }
      if (get_sid() == SID && warp_id == WID) {
        for (unsigned i = 0; i < m_call_record[warp_id].size(); i++) {
          printf("m_call_record[%u][%u]: <0x%llx %llx>\n", warp_id, i, 
                  m_call_record[warp_id][i].first, m_call_record[warp_id][i].second.to_ullong());
        }
      }
      
      // fflush(stdout);
      return true; // Ni: start spills&fills after regs exhausted
    }
    else if (pI->m_opcode == OP_RET) {
      bool flag = true;
      active_mask_t mask_temp = pI->get_active_mask();
      // if (get_sid() == SID && warp_id == WID)
      //     printf("pI mask %llx, record mask %llx\n", mask_temp.to_ullong(),
      //             m_call_record[warp_id].back().second);
      if (mask_temp.any()) {
        for (int j = m_call_record[warp_id].size()-1; j >= 0; j--) {
          bool ret_true = (m_call_record[warp_id][j].second & mask_temp).any();
          if (ret_true) {
            std::string mask1 = m_call_record[warp_id][j].second.to_string();
            std::string mask2 = mask_temp.to_string();
            if (mask1 >= mask2) {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
              // m_call_record[warp_id][j].second = m_call_record[warp_id][j].second ^ mask_temp; // bug
              // mask_temp.reset();
            }
            else {
              for (unsigned k = 0; k < WARP_SIZE; k++) {
                if (mask_temp.test(k) && m_call_record[warp_id][j].second.test(k)) {
                  mask_temp.reset(k);
                  m_call_record[warp_id][j].second.reset(k);
                }
              }
            }
            if (mask_temp.none()) {
              break;
            }
          }
        }
      }

      while (m_call_record[warp_id].back().second == 0) {
        if (get_sid() == SID && warp_id == WID)
          printf("pop 0x%llx\n", m_call_record[warp_id].back().first);
        m_call_record[warp_id].pop_back();
        if (!m_pair_record[warp_id].back()) {
          m_free_reg_per_warp[warp_id] += reg_win;
        }
        m_pair_record[warp_id].pop_back();
      }
      fflush(stdout);
      return true;
    }
    else {
      return true;
    }
    
    return true; 
  }
  else {
    if (pI->m_opcode == OP_CALL && pI->m_is_relo_call) {
      if (m_free_reg_number >= reg_win) {
        m_free_reg_number -= reg_win;
        // printf("Reg call %d %d!\n", m_free_reg_number, reg_win);
        // printf("Reserve %u number of registers at cycle %llu on SM %u\n", reg_win, curr_cycle, get_sid());
        return true;
      } else {
        // printf("Full reserve %u number of registers at cycle %llu on SM %u\n", reg_win, curr_cycle, get_sid());
        return false;
        // return true;
      }
    } else if (pI->m_opcode == OP_RET) {
      m_free_reg_number += reg_win;
      // printf("Reg return %d %d!\n", m_free_reg_number, reg_win);
      // printf("Release %u number of registers at cycle %llu on SM %u\n", reg_win, curr_cycle, get_sid());
      return true;
    } else {
      return true;
    }
    // Will never calls this
    return false;
  }
  // Will never calls this
  return false;
    return true;
}
